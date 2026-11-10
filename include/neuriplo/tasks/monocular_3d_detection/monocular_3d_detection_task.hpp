#pragma once

#include "neuriplo/tasks/core/base_task.hpp"
#include "neuriplo/tasks/core/preprocessor.hpp"
#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_postprocessor.hpp"

#include <memory>
#include <string>

namespace neuriplo_tasks {

/**
 * @brief Monocular 3D object-detection task (UrbanOmniDetect style).
 *
 * Single road-scene image in, projected cuboid detections out. Preprocessing
 * reuses the YOLO letterbox ([0,1], BGR->RGB, NCHW float32); the postprocessor
 * inverts it for both boxes and cuboid corners.
 */
class Monocular3DDetectionTask : public BaseTask {
  public:
    enum class ModelType : uint8_t { URBANOMNI, URBANOMNI_V2 };

    explicit Monocular3DDetectionTask(const ModelInfo& model_info, const std::string& model_name,
                                      float confidence_threshold = 0.25f, float nms_threshold = 0.45f);
    ~Monocular3DDetectionTask() override;

    TaskType getTaskType() override { return TaskType::Monocular3DDetection; }

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<Monocular3DPostprocessor> postprocessor_;

    [[nodiscard]] const Preprocessor& getPreprocessor() const override;
    std::vector<Result> decode(const Size& frame_size, const std::vector<Tensor>& tensors) override;

    static ModelType detectModelType(const std::string& model_name);
    static Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks
