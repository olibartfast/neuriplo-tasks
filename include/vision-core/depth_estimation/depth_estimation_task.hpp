#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"

#include <memory>
#include <string>

namespace vision_core {

class DepthEstimationPostprocessor;

/**
 * @brief Unified depth estimation task
 */
class DepthEstimationTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t {
        DEPTH_ANYTHING_V2
    };

    explicit DepthEstimationTask(const ModelInfo& model_info, const std::string& model_name);

    TaskType getTaskType() override { return TaskType::DepthEstimation; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<DepthEstimationPostprocessor> postprocessor_;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    std::unique_ptr<DepthEstimationPostprocessor> createPostprocessor(ModelType type);
    cv::Size extractInputSize(const ModelInfo& model_info);
};

} // namespace vision_core
