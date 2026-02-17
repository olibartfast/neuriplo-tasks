#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"

#include <memory>

namespace vision_core {

/**
 * @brief Base class for detection tasks
 *
 * Implements common logic for detection tasks (preprocessing, threshold management)
 * to reduce boilerplate in concrete implementations.
 */
class BaseDetectionTask : public TaskInterface {
  public:
    BaseDetectionTask(const ModelInfo& model_info, float conf_thresh = 0.25f, float nms_thresh = 0.45f)
        : TaskInterface(model_info), confidence_threshold_(conf_thresh), nms_threshold_(nms_thresh) {}

    TaskType getTaskType() override { return TaskType::Detection; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        if (!preprocessor_) {
            throw std::runtime_error("Preprocessor not initialized");
        }
        return preprocessor_->preprocess(imgs);
    }

  protected:
    std::unique_ptr<Preprocessor> preprocessor_;
    float confidence_threshold_;
    float nms_threshold_;
};

} // namespace vision_core
