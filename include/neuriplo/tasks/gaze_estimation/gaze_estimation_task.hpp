#pragma once

#include "neuriplo/tasks/core/base_task.hpp"
#include "neuriplo/tasks/core/preprocessor.hpp"
#include "neuriplo/tasks/gaze_estimation/gaze_estimation_postprocessor.hpp"

#include <memory>
#include <string>

namespace neuriplo_tasks {

/**
 * @brief Appearance-based gaze estimation task (ETH-XGaze / GazeTR style).
 *
 * Input is a normalized face or eye crop; face detection and calibration stay
 * outside this contract (see docs/gaze_estimation.md). Both routed families
 * share the same decoder: [B,2] pitch/yaw radians or [B,3] gaze vector.
 */
class GazeEstimationTask : public BaseTask {
  public:
    enum class ModelType : uint8_t { ETHXGAZE, GAZETR };

    explicit GazeEstimationTask(const ModelInfo& model_info, const std::string& model_name);
    ~GazeEstimationTask() override;

    TaskType getTaskType() override { return TaskType::GazeEstimation; }

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<GazePostprocessor> postprocessor_;

    [[nodiscard]] const Preprocessor& getPreprocessor() const override;
    std::vector<Result> decode(const Size& frame_size, const std::vector<Tensor>& tensors) override;

    static ModelType detectModelType(const std::string& model_name);
    static Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks
