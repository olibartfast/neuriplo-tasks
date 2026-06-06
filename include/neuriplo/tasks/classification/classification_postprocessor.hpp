#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for classification postprocessing
 */
class ClassificationPostprocessor {
  public:
    virtual ~ClassificationPostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param output Inference output tensor
     * @param shape Output tensor shape
     * @return Vector of Classification results
     */
    virtual std::vector<Classification> postprocess(const std::vector<TensorElement>& output,
                                                    const std::vector<int64_t>& shape) = 0;
};

/**
 * @brief Default implementation for standard classification models
 */
class DefaultClassificationPostprocessor : public ClassificationPostprocessor {
  public:
    DefaultClassificationPostprocessor(int top_k, bool apply_softmax);

    std::vector<Classification> postprocess(const std::vector<TensorElement>& output,
                                            const std::vector<int64_t>& shape) override;

  private:
    int top_k_;
    bool apply_softmax_;

    void applySoftmax(std::vector<float>& logits);
};

/**
 * @brief Shared logits postprocess for Torchvision / ViT / TensorFlow / default paths.
 *
 * When the leading batch dimension is greater than 1, returns one top-1 class per
 * batch index (so `results.size() == N`). For `N == 1`, returns up to `top_k` classes.
 */
[[nodiscard]] std::vector<Classification> postprocessClassificationLogits(const std::vector<TensorElement>& output,
                                                                          const std::vector<int64_t>& shape, int top_k,
                                                                          bool apply_softmax);

} // namespace neuriplo_tasks
