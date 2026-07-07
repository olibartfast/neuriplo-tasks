#pragma once

#include "neuriplo/tasks/core/base_task.hpp"
#include "neuriplo/tasks/core/preprocessor.hpp"

#include <memory>
#include <string>

namespace neuriplo_tasks {

class DepthEstimationPostprocessor;

/**
 * @brief Unified depth estimation task
 */
class DepthEstimationTask : public BaseTask {
  public:
    enum class ModelType : uint8_t { DEPTH_ANYTHING_V2 };

    explicit DepthEstimationTask(const ModelInfo& model_info, const std::string& model_name);
    ~DepthEstimationTask() override;

    TaskType getTaskType() override { return TaskType::DepthEstimation; }

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<DepthEstimationPostprocessor> postprocessor_;

    [[nodiscard]] const Preprocessor& getPreprocessor() const override;
    std::vector<Result> decode(const Size& frame_size, const std::vector<Tensor>& tensors) override;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const Size& input_size);
    std::unique_ptr<DepthEstimationPostprocessor> createPostprocessor(ModelType type);
    Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks
