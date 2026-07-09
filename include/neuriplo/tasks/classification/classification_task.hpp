#pragma once

#include "neuriplo/tasks/core/base_task.hpp"
#include "neuriplo/tasks/core/preprocessor.hpp"

#include <memory>
#include <string>

namespace neuriplo_tasks {

class ClassificationPostprocessor;

/**
 * @brief Unified classification task for all classifier architectures
 *
 * Handles all classification model types with consistent architecture:
 * - Torchvision classifiers (ResNet, etc.) with ImageNet normalization
 * - TensorFlow classifiers with [0,1] normalization
 * - Vision Transformer (ViT) classifiers
 *
 * Design principles:
 * 1. Single task class for all classification models
 * 2. Model type enum based on preprocessing requirements
 * 3. Factory pattern for preprocessor creation
 * 4. Unified postprocessing with configurable parameters
 */
class ClassificationTask : public BaseTask {
  public:
    /**
     * @brief Classification model types grouped by preprocessing requirements
     */
    enum class ModelType : uint8_t {
        TORCHVISION, // ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        TENSORFLOW,  // [0,1] normalization only
        VIT          // ViT-specific preprocessing
    };

    /**
     * @brief Create classification task for specified model type
     */
    explicit ClassificationTask(const ModelInfo& model_info, const std::string& model_name, int top_k = 5,
                                bool apply_softmax = true);
    ~ClassificationTask() override;

    TaskType getTaskType() override { return TaskType::Classification; }

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<ClassificationPostprocessor> postprocessor_;
    int top_k_;
    bool apply_softmax_;

    [[nodiscard]] const Preprocessor& getPreprocessor() const override;
    std::vector<Result> decode(const Size& frame_size, const std::vector<Tensor>& tensors) override;

    /**
     * @brief Detect model type from model name string
     */
    static ModelType detectModelType(const std::string& model_name);

    /**
     * @brief Create appropriate preprocessor for model type
     */
    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const Size& input_size);

    /**
     * @brief Create appropriate postprocessor for model type
     */
    std::unique_ptr<ClassificationPostprocessor> createPostprocessor(ModelType type);

    /**
     * @brief Extract input dimensions from model info
     */
    Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks
